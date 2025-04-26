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
    void testToString_allFieldsPopulated() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        details.setSellerId("12345");
        details.setSellerNickname("JohnDoe");
        details.setExchangeId("exch123");
        details.setOfferingPrice("$100");
        details.setCondition("Used");
        details.setConditionType("Good");
        details.setExchangeAvailability("Yes");
        details.setSellerCountry("USA");
        details.setSellerState("CA");
        details.setShipComments("Ships fast");
        details.setSellerRating("4.5");
        String expectedOutput = "------------------- \n" + "SellerId = 12345\n" + "SellerNickName = JohnDoe\n" + "ExchangeID = exch123\n" + "Price =  $100\n" + "Condition = Used\n" + "Condition Type = Good\n" + "Exchange Availability = Yes\n" + "County = USA\n" + "State = CA\n" + "Comments = Ships fast\n" + "------------------- \n";
        assertEquals(expectedOutput, details.toString());
    }

    @Test
    void testToString_emptyFields() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        String expectedOutput = "------------------- \n" + "SellerId = null\n" + "SellerNickName = null\n" + "ExchangeID = null\n" + "Price =  null\n" + "Condition = null\n" + "Condition Type = null\n" + "Exchange Availability = null\n" + "County = null\n" + "State = null\n" + "Comments = null\n" + "------------------- \n";
        assertEquals(expectedOutput, details.toString());
    }

    @Test
    void testToString_someFieldsPopulated() {
        ThirdPartyProductDetails details = new ThirdPartyProductDetails();
        details.setSellerId("12345");
        details.setSellerNickname("JaneDoe");
        details.setOfferingPrice("$50");
        String expectedOutput = "------------------- \n" + "SellerId = 12345\n" + "SellerNickName = JaneDoe\n" + "ExchangeID = null\n" + "Price =  $50\n" + "Condition = null\n" + "Condition Type = null\n" + "Exchange Availability = null\n" + "County = null\n" + "State = null\n" + "Comments = null\n" + "------------------- \n";
        assertEquals(expectedOutput, details.toString());
    }
}
