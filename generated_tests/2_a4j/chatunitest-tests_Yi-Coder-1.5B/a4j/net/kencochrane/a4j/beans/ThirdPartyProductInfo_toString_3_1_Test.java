package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class ThirdPartyProductInfo_toString_3_1_Test {

    @Test
    public void testToString() {
        ThirdPartyProductInfo thirdPartyProductInfo = new ThirdPartyProductInfo();
        String expected = "productOffers is null\n# of productOffers = 0";
        assertEquals(expected, thirdPartyProductInfo.toString());
    }
}
