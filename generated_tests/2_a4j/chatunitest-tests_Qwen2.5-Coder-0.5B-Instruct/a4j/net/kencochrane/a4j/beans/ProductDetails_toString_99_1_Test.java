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

    @Mock
    private ProductDetails productDetails;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetSavings() {
        // Arrange
        String expectedSavings = "You save $10 that's 5% off the list price!";
        when(productDetails.getSavings()).thenReturn(expectedSavings);
        // Act
        String actualSavings = productDetails.getSavings();
        // Assert
        assertEquals(expectedSavings, actualSavings);
    }
}
