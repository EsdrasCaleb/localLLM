// ThirdPartyProductInfo_toString_3_1_Test.java
package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

public class ThirdPartyProductInfo_toString_3_1_Test {

    @Test
    public void testToString() {
        ThirdPartyProductInfo productInfo = mock(ThirdPartyProductInfo.class);
        ThirdPartyProductDetails[] productDetails = new ThirdPartyProductDetails[] { new ThirdPartyProductDetails() };
        when(productInfo.getThirdPartyProductDetails()).thenReturn(productDetails);
        assertEquals("ThirdPartyProductDetails\n" + "ThirdPartyProductDetails\n" + "# of productOffers = 1", productInfo.toString());
        verify(productInfo).getThirdPartyProductDetails();
    }
}
