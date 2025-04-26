package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ThirdPartyProductInfo_toString_3_0_Test {

    @Test
    public void testToString() throws Exception {
        ThirdPartyProductInfo thirdPartyProductInfo = new ThirdPartyProductInfo();
        Field field = ThirdPartyProductInfo.class.getDeclaredField("productOffers");
        field.setAccessible(true);
        // Mock the productOffers field
        ArrayList mockProductOffers = new ArrayList();
        mockProductOffers.add(new ThirdPartyProductDetails());
        mockProductOffers.add(new ThirdPartyProductDetails());
        field.set(thirdPartyProductInfo, mockProductOffers);
        // Call the toString method
        String result = thirdPartyProductInfo.toString();
        // Check the result
        assertEquals("ThirdPartyProductDetails{productDetails=null, offerId=null, offerName=null, offerPrice=null, offerQuantity=null}\n" + "ThirdPartyProductDetails{productDetails=null, offerId=null, offerName=null, offerPrice=null, offerQuantity=null}\n" + "# of productOffers = 2", result);
    }
}
