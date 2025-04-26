package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class ThirdPartyProductDetails_toString_22_1_Test {

    @Test
    void testToString() {
        ThirdPartyProductDetails product = new ThirdPartyProductDetails();
        product.setSellerId("12345");
        product.setSellerNickname("John Doe");
        product.setExchangeId("ABC123");
        product.setOfferingPrice("100.00");
        product.setCondition("New");
        product.setConditionType("New");
        product.setExchangeAvailability("Available");
        product.setSellerCountry("USA");
        product.setSellerState("CA");
        product.setShipComments("Free shipping");
        product.setSellerRating("5");
        String expectedOutput = "-------------------\n" + "SellerId = 12345\n" + "SellerNickName = John Doe\n" + "ExchangeID = ABC123\n" + "Price = 100.00\n" + "Condition = New\n" + "Condition Type = New\n" + "ExchangeAvailability = Available\n" + "County = USA\n" + "State = CA\n" + "Comments = Free shipping\n" + "-------------------\n";
        assertEquals(expectedOutput, product.toString());
    }
}
