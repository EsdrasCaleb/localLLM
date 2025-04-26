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
    void testGetAsin() {
        // Given
        ProductDetails productDetails = new ProductDetails();
        String asin = "1234567890";
        // When
        String asinResult = productDetails.getAsin();
        // Then
        assertEquals(asin, asinResult);
    }
}
