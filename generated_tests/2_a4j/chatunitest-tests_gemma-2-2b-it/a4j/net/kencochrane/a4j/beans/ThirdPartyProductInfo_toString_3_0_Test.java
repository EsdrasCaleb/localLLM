package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ThirdPartyProductInfo_toString_3_0_Test {

    @Test
    void testToString() {
        ThirdPartyProductInfo productInfo = new ThirdPartyProductInfo();
        productInfo.setThirdPartyProductDetails(new ThirdPartyProductDetails[] { new ThirdPartyProductDetails(), new ThirdPartyProductDetails() });
        String expectedOutput = "productOffers is null \n# of productOffers = 0";
        String actualOutput = productInfo.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
