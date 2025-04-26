package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.math.BigDecimal;
import java.text.DecimalFormat;

public class ProductDetails_toString_99_1_Test {

    @InjectMocks
    private ProductDetails productDetails;

    @Mock
    private Tracks tracks;

    @Mock
    private Lists lists;

    @Mock
    private Artists artists;

    @Mock
    private Features features;

    @Mock
    private SimilarProducts similarProducts;

    @Mock
    private Reviews reviews;

    @Mock
    private BrowseList browseList;

    @Mock
    private Accessories accessories;

    @Mock
    private Directors directors;

    @Mock
    private Starring starring;

    @Mock
    private Authors authors;

    @Mock
    private Platforms platforms;

    @Mock
    private ThirdPartyProductInfo thirdPartyProductInfo;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetSavings() {
        productDetails.setListPrice("100.00");
        productDetails.setOurPrice("80.00");
        String savings = productDetails.getSavings();
        assertEquals(" (You save $20.00 that's 20.0% off the list price!)", savings);
    }

    @Test
    public void testGetSavingsNullListPrice() {
        productDetails.setListPrice(null);
        productDetails.setOurPrice("80.00");
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavingsNullOurPrice() {
        productDetails.setListPrice("100.00");
        productDetails.setOurPrice(null);
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetRatingsImgURL() {
        when(reviews.getAvgCustomerRating()).thenReturn("4.5");
        String url = productDetails.getRatingsImgURL();
        assertEquals("/images/stars-4.gif", url);
    }

    @Test
    public void testGetRatingsImgURLInvalidRating() {
        when(reviews.getAvgCustomerRating()).thenReturn("6");
        String url = productDetails.getRatingsImgURL();
        assertNull(url);
    }

    @Test
    public void testGetRatingsImgURLNullAvgCustomerRating() {
        when(reviews.getAvgCustomerRating()).thenReturn(null);
        String url = productDetails.getRatingsImgURL();
        assertNull(url);
    }

    @Test
    public void testGetRecommendation() {
        when(reviews.getAvgCustomerRating()).thenReturn("4.5");
        String recommendation = productDetails.getRecommendation();
        assertEquals("80.0% of our Customers recommend this Product!", recommendation);
    }

    @Test
    public void testGetRecommendationInvalidRating() {
        when(reviews.getAvgCustomerRating()).thenReturn("6");
        String recommendation = productDetails.getRecommendation();
        assertNull(recommendation);
    }

    @Test
    public void testGetRecommendationNullAvgCustomerRating() {
        when(reviews.getAvgCustomerRating()).thenReturn(null);
        String recommendation = productDetails.getRecommendation();
        assertNull(recommendation);
    }
}
