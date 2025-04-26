package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.math.BigDecimal;
import java.text.DecimalFormat;

@ExtendWith(MockitoExtension.class)
public class ProductDetails_toString_99_0_Test {

    @InjectMocks
    private ProductDetails productDetails;

    @BeforeEach
    public void setup() {
        productDetails.setListPrice("10.99");
        productDetails.setOurPrice("8.99");
    }

    @Test
    public void testGetSavings() {
        String savings = productDetails.getSavings();
        assertEquals(" (You save $1.00 that's 90.00% off the list price!)", savings);
    }

    @Test
    public void testGetSavings_NullListPrice() {
        productDetails.setListPrice(null);
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavings_NullOurPrice() {
        productDetails.setOurPrice(null);
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavings_ZeroListPrice() {
        productDetails.setListPrice("0.00");
        productDetails.setOurPrice("0.00");
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavings_ZeroOurPrice() {
        productDetails.setListPrice("10.99");
        productDetails.setOurPrice("0.00");
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavings_NonDecimalListPrice() {
        productDetails.setListPrice("10");
        productDetails.setOurPrice("8");
        String savings = productDetails.getSavings();
        assertNull(savings);
    }

    @Test
    public void testGetSavings_NonDecimalOurPrice() {
        productDetails.setListPrice("10.99");
        productDetails.setOurPrice("8");
        String savings = productDetails.getSavings();
        assertNull(savings);
    }
}
