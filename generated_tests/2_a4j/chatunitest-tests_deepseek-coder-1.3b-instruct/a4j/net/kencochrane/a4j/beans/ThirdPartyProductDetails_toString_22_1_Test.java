package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

class ThirdPartyProductDetails_toString_22_1_Test {

    @Test
    void testToString() {
        ThirdPartyProductDetails product = new ThirdPartyProductDetails();
        product.setSellerId("123");
        product.setSellerNickname("TestNick");
        product.setExchangeId("456");
        product.setOfferingPrice("789");
        product.setCondition("Good");
        product.setConditionType("New");
        product.setExchangeAvailability("Available");
        product.setSellerCountry("USA");
        product.setSellerState("NY");
        product.setShipComments("No Comments");
        product.setSellerRating("5");
        String expected = "------------------- \n" + "SellerId = 123\n" + "SellerNickName = TestNick\n" + "ExchangeID = 456\n" + "Price =  789\n" + "Condition =  Good\n" + "Condition Type =  New\n" + "Exchange Availability =  Available\n" + "County =  USA\n" + "State =  NY\n" + "Comments =  No Comments\n" + "------------------- \n";
        assertEquals(expected, product.toString());
    }
}
