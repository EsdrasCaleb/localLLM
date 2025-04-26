// ThirdPartyProductInfo_toString_3_1_Test.java
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ThirdPartyProductInfo_toString_3_1_Test {

    @Mock
    private ThirdPartyProductDetails[] products;

    @InjectMocks
    private ThirdPartyProductInfo focal;

    @BeforeEach
    public void setup() {
        when(focal.getProductsArrayList()).thenReturn(new ArrayList<>());
    }

    @Test
    public void testToString() {
        // Arrange
        when(focal.getProductsArrayList()).thenReturn(new ArrayList<>());
        // Act
        String result = focal.toString();
        // Assert
        assertEquals("[]", result);
    }

    @Test
    public void testToString_NoProducts() {
        // Arrange
        when(focal.getProductsArrayList()).thenReturn(null);
        // Act
        String result = focal.toString();
        // Assert
        assertNull(result);
    }
}
