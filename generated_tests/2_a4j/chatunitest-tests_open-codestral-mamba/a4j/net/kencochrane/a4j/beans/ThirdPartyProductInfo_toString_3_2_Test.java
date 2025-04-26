package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ThirdPartyProductInfo_toString_3_2_Test {

    @Mock
    private ArrayList productOffers;

    @InjectMocks
    private ThirdPartyProductInfo thirdPartyProductInfo;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToStringWithArrayListNotNull() {
        // Arrange
        ThirdPartyProductDetails productDetails1 = new ThirdPartyProductDetails();
        ThirdPartyProductDetails productDetails2 = new ThirdPartyProductDetails();
        when(productOffers.size()).thenReturn(2);
        when(productOffers.get(0)).thenReturn(productDetails1);
        when(productOffers.get(1)).thenReturn(productDetails2);
        String expectedOutput = productDetails1.toString() + "\n" + productDetails2.toString() + "\n# of productOffers = 2";
        // Act
        String actualOutput = thirdPartyProductInfo.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToStringWithArrayListNull() {
        // Arrange
        when(productOffers.size()).thenReturn(0);
        when(productOffers.get(0)).thenReturn(null);
        String expectedOutput = "productOffers is null ";
        // Act
        String actualOutput = thirdPartyProductInfo.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
