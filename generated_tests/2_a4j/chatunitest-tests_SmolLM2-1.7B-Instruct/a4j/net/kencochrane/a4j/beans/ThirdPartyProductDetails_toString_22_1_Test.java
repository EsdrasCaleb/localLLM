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
    public void testToString() {
        ThirdPartyProductDetails product = new ThirdPartyProductDetails();
        product.setSellerId("sellerId");
        product.setSellerNickname("sellerNickname");
        product.setExchangeId("exchangeId");
        product.setOfferingPrice("offeringPrice");
        product.setCondition("condition");
        product.setConditionType("conditionType");
        product.setExchangeAvailability("exchangeAvailability");
        product.setSellerCountry("sellerCountry");
        product.setSellerState("sellerState");
        product.setShipComments("shipComments");
        product.setSellerRating("sellerRating");
        String expected = "------------------- \n" + "SellerId = sellerId\n" + "SellerNickName = sellerNickname\n" + "ExchangeID = exchangeId\n" + "Price =  offeringPrice\n" + "Condition = condition\n" + "Condition Type = conditionType\n" + "Exchange Availability = exchangeAvailability\n" + "County = sellerCountry\n" + "State = sellerState\n" + "Comments = shipComments\n" + "------------------- \n";
        assertEquals(expected, product.toString());
    }
}
