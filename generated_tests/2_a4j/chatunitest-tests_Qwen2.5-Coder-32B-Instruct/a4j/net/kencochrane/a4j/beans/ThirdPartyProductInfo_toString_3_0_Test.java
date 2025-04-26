package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ThirdPartyProductInfo_toString_3_0_Test {

    @InjectMocks
    private ThirdPartyProductInfo thirdPartyProductInfo;

    @Mock
    private ArrayList<ThirdPartyProductDetails> productOffersMock;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString_ProductOffersIsNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field productOffersField = ThirdPartyProductInfo.class.getDeclaredField("productOffers");
        productOffersField.setAccessible(true);
        productOffersField.set(thirdPartyProductInfo, null);
        // Act
        String result = thirdPartyProductInfo.toString();
        // Assert
        assertEquals("productOffers is null ", result);
    }

    @Test
    public void testToString_ProductOffersIsEmpty() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field productOffersField = ThirdPartyProductInfo.class.getDeclaredField("productOffers");
        productOffersField.setAccessible(true);
        productOffersField.set(thirdPartyProductInfo, new ArrayList<>());
        // Act
        String result = thirdPartyProductInfo.toString();
        // Assert
        assertEquals("# of productOffers = 0", result);
    }

    @Test
    public void testToString_ProductOffersContainsElements() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ThirdPartyProductDetails product1 = mock(ThirdPartyProductDetails.class);
        ThirdPartyProductDetails product2 = mock(ThirdPartyProductDetails.class);
        when(product1.toString()).thenReturn("Product 1 Details");
        when(product2.toString()).thenReturn("Product 2 Details");
        ArrayList<ThirdPartyProductDetails> productOffers = new ArrayList<>();
        productOffers.add(product1);
        productOffers.add(product2);
        Field productOffersField = ThirdPartyProductInfo.class.getDeclaredField("productOffers");
        productOffersField.setAccessible(true);
        productOffersField.set(thirdPartyProductInfo, productOffers);
        // Act
        String result = thirdPartyProductInfo.toString();
        // Assert
        assertEquals("Product 1 Details\nProduct 2 Details\n# of productOffers = 2", result);
    }
}
