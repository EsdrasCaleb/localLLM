package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

class ThirdPartyProductDetails_toString_22_0_Test {

    @Test
    void testToString_allFieldsNotNull() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        details.setSellerId("123");
        details.setSellerNickname("SellerName");
        details.setExchangeId("456");
        details.setOfferingPrice("10.00");
        details.setCondition("New");
        details.setConditionType("Used");
        details.setExchangeAvailability("Available");
        details.setSellerCountry("USA");
        details.setSellerState("CA");
        details.setShipComments("Fast shipping");
        String expectedOutput = "------------------- \n" + "SellerId = 123\n" + "SellerNickName = SellerName\n" + "ExchangeID = 456\n" + "Price =  10.00\n" + "Condition = New\n" + "Condition Type = Used\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = CA\n" + "Comments = Fast shipping\n" + "------------------- \n";
        String actualOutput = details.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_someFieldsNull() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        details.setSellerId("123");
        details.setSellerNickname(null);
        details.setExchangeId("456");
        details.setOfferingPrice(null);
        details.setCondition("New");
        details.setConditionType(null);
        details.setExchangeAvailability("Available");
        details.setSellerCountry(null);
        details.setSellerState(null);
        details.setShipComments("Fast shipping");
        String expectedOutput = "------------------- \n" + "SellerId = 123\n" + "SellerNickName = null\n" + "ExchangeID = 456\n" + "Price =  null\n" + "Condition = New\n" + "Condition Type = null\n" + "Exchange Availability = Available\n" + "County = null\n" + "State = null\n" + "Comments = Fast shipping\n" + "------------------- \n";
        String actualOutput = details.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_allFieldsNull() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        String expectedOutput = "------------------- \n" + "SellerId = null\n" + "SellerNickName = null\n" + "ExchangeID = null\n" + "Price =  null\n" + "Condition = null\n" + "Condition Type = null\n" + "Exchange Availability = null\n" + "County = null\n" + "State = null\n" + "Comments = null\n" + "------------------- \n";
        String actualOutput = details.toString();
        Assertions.assertEquals(expectedOutput, actualOutput);
    }
}
