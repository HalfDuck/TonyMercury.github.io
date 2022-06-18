$(window).load(function() {
    "use strict";
    /*=========================================================================
     Wow Initialize
     =========================================================================*/
    // Here will be the WoW Js implementation.
    setTimeout(function(){new WOW().init();}, 0);

    /*=========================================================================
     Isotope
     =========================================================================*/
    $('.portfolio-filter').on( 'click', 'li', function() {
        var filterValue = $(this).attr('data-filter');
        $grid.isotope({ filter: filterValue });
    });

    // change is-checked class on buttons
    $('.portfolio-filter').each( function( i, buttonGroup ) {
        var $buttonGroup = $( buttonGroup );
        $buttonGroup.on( 'click', 'li', function() {
            $buttonGroup.find('.current').removeClass('current');
            $( this ).addClass('current');
        });
    });

    var $grid = $('.portfolio-wrapper').isotope({
        // options
        itemSelector: '[class*="col-"]',
        masonry: {
            // use element for option
            columnWidth: '[class*="col-"]'
        }
    });

    /* ======= Mobile Filter ======= */

    // bind filter on select change
    $('.portfolio-filter-mobile').on( 'change', function() {
      // get filter value from option value
      var filterValue = this.value;
      // use filterFn if matches value
      filterValue = filterFns[ filterValue ] || filterValue;
      $grid.isotope({ filter: filterValue });
    });

    var filterFns = {
      // show if number is greater than 50
      numberGreaterThan50: function() {
        var number = $(this).find('.number').text();
        return parseInt( number, 10 ) > 50;
      },
      // show if name ends with -ium
      ium: function() {
        var name = $(this).find('.name').text();
        return name.match( /ium$/ );
      }
    };

});

$(document).ready(function() {
    "use strict";

    // Preloader
    $('body').jpreLoader();

    /*=========================================================================
     Clients carousel
     =========================================================================*/
    $('.clients-wrapper').slick({
      dots: false,
      arrows: false,
      slidesToShow: 5,
      slidesToScroll: 1,
      autoplay: true,
      autoplaySpeed: 2000,
      responsive: [
        {
          breakpoint: 767,
          settings: {
            slidesToShow: 3,
            slidesToScroll: 3,
          }
        },
        {
          breakpoint: 515,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1,
          }
        }
      ]
    });

    /*=========================================================================
     Services carousel
     =========================================================================*/
    $('.services-wrapper').slick({
      dots: false,
      arrows: true,
      slidesToShow: 3,
      slidesToScroll: 1,
      autoplay: true,
      autoplaySpeed: 2000,
      responsive: [
        {
          breakpoint: 767,
          settings: {
            slidesToShow: 2,
            slidesToScroll: 2,
          }
        },
        {
          breakpoint: 515,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1,
          }
        }
      ]
    });

    /*=========================================================================
     Testimonials carousel
     =========================================================================*/
    $('.testimonials-wrapper').slick({
      dots: true,
      arrows: false,
      slidesToShow: 1,
      slidesToScroll: 1,
    });

    /*=========================================================================
     One Page Scroll with jQuery
     =========================================================================*/
    $('.horizontal-menu ul li a[href^="#"]:not([href="#"]), .mobile-menu ul li a[href^="#"]:not([href="#"]), .scroll-down-icon').on('click', function(event) {
      var $anchor = $(this);
      $('html, body').stop().animate({
        scrollTop: $($anchor.attr('href')).offset().top
      }, 800, 'easeInOutQuad');
      event.preventDefault();
    });

    /*=========================================================================
     Add (nav-link) class to main & mobile menu.
     =========================================================================*/
    $('.horizontal-menu ul li a, .mobile-menu ul li a').addClass('nav-link');

});

$(function(){
    "use strict";

    /*=========================================================================
     Bootstrap Scrollspy
     =========================================================================*/
    $("body").scrollspy({ target: ".scrollspy"});

    /*=========================================================================
     Mobile Menu
     =========================================================================*/
    $('.menu-toggle').on( 'click', function() {
        $('.mobile-menu').slideToggle(300);
    });

    $('.mobile-menu').on( 'click', 'li', 'a', function() {
        $('.mobile-menu').slideToggle(300);
    });

    /*=========================================================================
     Fixing alignment of clients logo
     =========================================================================*/
    $('.clients-wrapper .slick-track').addClass('d-flex align-items-center');

    /*=========================================================================
     Parallax layers
     =========================================================================*/
    if ($('.parallax').length > 0) { 
      var scene = $('.parallax').get(0);
      var parallax = new Parallax(scene, { 
        relativeInput: true,
        onReady: function() { console.log('ready!');
      } });
    }

    /*=========================================================================
     Parallax background images with Stellar JS
     =========================================================================*/
    $.stellar({
      horizontalScrolling: false,
      verticalOffset: 40
    });

    /*=========================================================================
     Progress bar animation with Waypoint JS
     =========================================================================*/
    if ($('.about').length > 0) { 
      var waypoint = new Waypoint({
        element: document.getElementsByClassName('about'),
        handler: function(direction) {
          
          $('.progress-bar').each(function() {
            var bar_value = $(this).attr('aria-valuenow') + '%';                
            $(this).animate({ width: bar_value }, { easing: 'linear' });
          });

          this.destroy()
        }
      });
    }

    /*=========================================================================
            Scroll to Top
    =========================================================================*/
    $(window).scroll(function() {
        if ($(this).scrollTop() >= 250) {        // If page is scrolled more than 50px
            $('#return-to-top').fadeIn(200);    // Fade in the arrow
        } else {
            $('#return-to-top').fadeOut(200);   // Else fade out the arrow
        }
    });
    $('#return-to-top').click(function() {      // When arrow is clicked
        $('body,html').animate({
            scrollTop : 0                       // Scroll to top of body
        }, 400);
    });

    /*=========================================================================
     Counterup JS for facts
     =========================================================================*/
    $('.count').counterUp({
      delay: 10,
      time: 2000
    });

    /*=========================================================================
     Single Portfolio Header Image with Data Attribute
     =========================================================================*/
    var list = document.getElementsByClassName('portfolio-header');

    for (var i = 0; i < list.length; i++) {
      var src = list[i].getAttribute('data-image-src');
      list[i].style.backgroundImage="url('" + src + "')";
    }

    /*=========================================================================
            Magnific Popup Functions
    =========================================================================*/
    $('.portfolio-image').magnificPopup({
      type: 'image',
      removalDelay: 500, //delay removal by X to allow out-animation
      callbacks: {
        beforeOpen: function() {
          // just a hack that adds mfp-anim class to markup 
           this.st.image.markup = this.st.image.markup.replace('mfp-figure', 'mfp-figure mfp-with-anim');
           this.st.mainClass = this.st.el.attr('data-effect');
        }
      },
      closeOnContentClick: true,
      midClick: true
    });

    $('.portfolio-video').magnificPopup({
      type: 'iframe',
      iframe: {
          markup: '<div class="mfp-iframe-scaler">'+
                    '<div class="mfp-close"></div>'+
                    '<iframe class="mfp-iframe" frameborder="0" allowfullscreen></iframe>'+
                  '</div>', 

          patterns: {
            youtube: {
              index: 'youtube.com/',

              id: 'v=',

              src: 'http://www.youtube.com/embed/%id%?autoplay=1'
            },
            vimeo: {
              index: 'vimeo.com/',
              id: '/',
              src: '//player.vimeo.com/video/%id%?autoplay=1'
            },
            gmaps: {
              index: '//maps.google.',
              src: '%id%&output=embed'
            }

          },

          srcAction: 'iframe_src',
        }
    });

});

/*=========================================================================
      Flat Surface Shader
=========================================================================*/
if ($('.shader').length > 0) { 
  
  // strict mode is not allowed

  var container = document.getElementById('hero');
  var renderer = new FSS.SVGRenderer();
  var scene = new FSS.Scene();
  var light = new FSS.Light('#060606', '#222222');
  var geometry = new FSS.Plane(screen.availWidth, screen.availHeight, 6, 4);
  var material = new FSS.Material('#FFFFFF', '#FFFFFF');
  var mesh = new FSS.Mesh(geometry, material);
  var now, start = Date.now();

  function initialise() {
    scene.add(mesh);
    scene.add(light);
    container.appendChild(renderer.element);
    window.addEventListener('resize', resize);
  }

  function resize() {
    renderer.setSize(container.offsetWidth, container.offsetHeight);
  }

  function animate() {
    now = Date.now() - start;
    light.setPosition(300*Math.sin(now*0.001), 200*Math.cos(now*0.0005), 60);
    renderer.render(scene);
    requestAnimationFrame(animate);
  }

  initialise();
  resize();
  animate();
}

/*=========================================================================
        Sticky Header Animation
=========================================================================*/
var animatedHeader = (function() {
  "use strict";

  var docElem = document.documentElement,
    header = document.querySelector( "header.over" ),
    didScroll = false,
    changeHeaderOn = 100;

  function init() {
    window.addEventListener( 'scroll', function( event ) {
      if( !didScroll ) {
        didScroll = true;
        setTimeout( scrollPage, 250 );
      }
    }, false );
  }

  function scrollPage() {
    var sy = scrollY();
    if ( sy >= changeHeaderOn ) {
      $(header).addClass('shrink');
    }
    else {
      $(header).removeClass('shrink');
    }
    didScroll = false;
  }

  function scrollY() {
    return window.pageYOffset || docElem.scrollTop;
  }

  init();

})();