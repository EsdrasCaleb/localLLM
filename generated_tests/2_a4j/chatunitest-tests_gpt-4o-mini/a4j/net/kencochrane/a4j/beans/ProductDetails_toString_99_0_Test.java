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

public class ProductDetails_toString_99_0_Test {

    private ProductDetails productDetails;

    private Reviews mockReviews;

    @BeforeEach
    public void setUp() {
        productDetails = new ProductDetails();
        mockReviews = mock(Reviews.class);
        productDetails.setReviews(mockReviews);
    }

    @Test
    public void testGetSavings_ValidPrices() {
        productDetails.setListPrice("$50.00");
        productDetails.setOurPrice("$30.00");
        String result = productDetails.getSavings();
        assertNotNull(result);
        assertTrue(result.contains("You save $20.00"));
        assertTrue(result.contains("that's 40.00% off"));
    }

    @Test
    public void testGetSavings_NullListPrice() {
        productDetails.setOurPrice("$30.00");
        String result = productDetails.getSavings();
        assertNull(result);
    }

    @Test
    public void testGetSavings_NullOurPrice() {
        productDetails.setListPrice("$50.00");
        String result = productDetails.getSavings();
        assertNull(result);
    }

    @Test
    public void testGetSavings_InvalidPrices() {
        productDetails.setListPrice("$50.00");
        productDetails.setOurPrice("invalid");
        String result = productDetails.getSavings();
        assertNull(result);
    }

    @Test
    public void testGetRatingsImgURL_ValidRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn("4.0");
        String result = productDetails.getRatingsImgURL();
        assertEquals("/images/stars-4.gif", result);
    }

    @Test
    public void testGetRatingsImgURL_InvalidRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn("6.0");
        String result = productDetails.getRatingsImgURL();
        assertNull(result);
    }

    @Test
    public void testGetRatingsImgURL_NullRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn(null);
        String result = productDetails.getRatingsImgURL();
        assertNull(result);
    }

    @Test
    public void testGetRecommendation_ValidRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn("3.0");
        String result = productDetails.getRecommendation();
        assertEquals("60% of our Customers recommend this Product!", result);
    }

    @Test
    public void testGetRecommendation_InvalidRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn("6.0");
        String result = productDetails.getRecommendation();
        assertNull(result);
    }

    @Test
    public void testGetRecommendation_NullRating() {
        when(mockReviews.getAvgCustomerRating()).thenReturn(null);
        String result = productDetails.getRecommendation();
        assertNull(result);
    }
}
