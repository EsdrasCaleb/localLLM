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

    @Test
    public void testGetAsin() {
        ProductDetails productDetails = new ProductDetails();
        String asin = productDetails.getAsin();
        assertEquals("asin", asin);
    }

    @Test
    public void testSetAsin() {
        ProductDetails productDetails = new ProductDetails();
        String asin = "B08Q2424QW";
        productDetails.setAsin(asin);
        assertEquals(asin, productDetails.getAsin());
    }
}
