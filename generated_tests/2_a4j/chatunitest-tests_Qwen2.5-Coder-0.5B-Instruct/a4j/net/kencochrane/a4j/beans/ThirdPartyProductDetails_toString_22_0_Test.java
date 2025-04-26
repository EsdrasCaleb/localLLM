package net.kencochrane.a4j.beans;

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
    public void testToString() {
        ThirdPartyProductDetails thirdPartyProductDetails = mock(ThirdPartyProductDetails.class);
        when(thirdPartyProductDetails.getSellerId()).thenReturn("123");
        when(thirdPartyProductDetails.getSellerNickname()).thenReturn("John Doe");
        when(thirdPartyProductDetails.getExchangeId()).thenReturn("ABC123");
        when(thirdPartyProductDetails.getOfferingPrice()).thenReturn("299.99");
        when(thirdPartyProductDetails.getCondition()).thenReturn("Good");
        when(thirdPartyProductDetails.getConditionType()).thenReturn("New");
        when(thirdPartyProductDetails.getExchangeAvailability()).thenReturn("Available");
        when(thirdPartyProductDetails.getSellerCountry()).thenReturn("USA");
        when(thirdPartyProductDetails.getSellerState()).thenReturn("CA");
        when(thirdPartyProductDetails.getShipComments()).thenReturn("Great deal!");
        String expectedOutput = "------------------- \n";
        expectedOutput += "SellerId = 123\n";
        expectedOutput += "SellerNickName = John Doe\n";
        expectedOutput += "ExchangeID = ABC123\n";
        expectedOutput += "Price = 299.99\n";
        expectedOutput += "Condition = Good\n";
        expectedOutput += "Condition Type = New\n";
        expectedOutput += "Exchange Availability = Available\n";
        expectedOutput += "County = USA\n";
        expectedOutput += "State = CA\n";
        expectedOutput += "Comments = Great deal!\n";
        expectedOutput += "------------------- \n";
        assertEquals(expectedOutput, thirdPartyProductDetails.toString());
    }
}
