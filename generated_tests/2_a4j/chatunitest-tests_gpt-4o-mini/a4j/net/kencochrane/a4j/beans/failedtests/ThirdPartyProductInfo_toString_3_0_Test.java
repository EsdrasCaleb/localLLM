package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ThirdPartyProductInfo_toString_3_0_Test {

    private ThirdPartyProductInfo productInfo;

    @BeforeEach
    public void setUp() {
        productInfo = new ThirdPartyProductInfo();
    }

    @Test
    public void testToString_WithProducts() {
        ThirdPartyProductDetails product1 = mock(ThirdPartyProductDetails.class);
        ThirdPartyProductDetails product2 = mock(ThirdPartyProductDetails.class);
        when(product1.toString()).thenReturn("Product 1");
        when(product2.toString()).thenReturn("Product 2");
        ThirdPartyProductDetails[] products = { product1, product2 };
        productInfo.setThirdPartyProductDetails(products);
        String expectedOutput = "Product 1\nProduct 2\n# of productOffers = 2";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    public void testToString_NoProducts() {
        productInfo.setThirdPartyProductDetails(new ThirdPartyProductDetails[0]);
        String expectedOutput = "# of productOffers = 0";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    public void testToString_ProductOffersNull() {
        productInfo.setThirdPartyProductDetails(null);
        String expectedOutput = "productOffers is null ";
        assertEquals(expectedOutput, productInfo.toString());
    }
}
