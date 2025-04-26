package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ThirdPartyProductDetails_toString_22_0_Test {

    // Test class
    @Test
    void testToString() {
        String input = "SellerId = 12345678901234567890123456789012\n" + "SellerNickName = Test\n" + "ExchangeID = 12345678901234567890123456789012\n" + "Price =  100000\n" + "Condition = New\n" + "Condition Type = Exchanged\n" + "Exchange Availability = In Stock\n" + "County = US\n" + "State = CA\n" + "Comments = Good condition\n" + "------------------- \n";
        ThirdPartyProductDetails product = new ThirdPartyProductDetails();
        product.setSellerId("12345678901234567890123456789012");
        product.setSellerNickname("Test");
        product.setExchangeId("12345678901234567890123456789012");
        product.setOfferingPrice("100000");
        product.setCondition("New");
        product.setConditionType("Exchanged");
        product.setExchangeAvailability("In Stock");
        product.setSellerCountry("US");
        product.setSellerState("CA");
        product.setShipComments("Good condition");
        assertEquals(input, product.toString());
    }
}
