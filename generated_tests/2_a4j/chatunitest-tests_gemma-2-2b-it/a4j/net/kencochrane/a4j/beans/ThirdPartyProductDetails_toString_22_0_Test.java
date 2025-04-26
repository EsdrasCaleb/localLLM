package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class ThirdPartyProductDetails_toString_22_0_Test {

    @Test
    void testToString() {
        ThirdPartyProductDetails productDetails = new ThirdPartyProductDetails();
        productDetails.setSellerId("123");
        productDetails.setSellerNickname("john_doe");
        productDetails.setExchangeId("456");
        productDetails.setOfferingPrice("100.00");
        productDetails.setCondition("used");
        productDetails.setConditionType("new");
        productDetails.setExchangeAvailability("available");
        productDetails.setSellerCountry("USA");
        productDetails.setSellerState("CA");
        productDetails.setShipComments("product is in good condition");
        productDetails.setSellerRating("5");
        String output = productDetails.toString();
        assertEquals("------------------- \nSellerId = 123\nSellerNickName = john_doe\nExchangeID = 456\nPrice =  100.00\nCondition = used\nCondition Type = new\nExchange Availability = available\nCounty = USA\nState = CA\nComments = product is in good condition\n------------------- \n", output);
    }
}
