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
    public void testToString() throws Exception {
        ThirdPartyProductDetails productDetails = Mockito.mock(ThirdPartyProductDetails.class);
        when(productDetails.getSellerId()).thenReturn("12345");
        when(productDetails.getSellerNickname()).thenReturn("seller");
        when(productDetails.getExchangeId()).thenReturn("abcde");
        when(productDetails.getOfferingPrice()).thenReturn("100$");
        when(productDetails.getCondition()).thenReturn("New");
        when(productDetails.getConditionType()).thenReturn("Used");
        when(productDetails.getExchangeAvailability()).thenReturn("Yes");
        when(productDetails.getSellerCountry()).thenReturn("USA");
        when(productDetails.getSellerState()).thenReturn("NY");
        when(productDetails.getShipComments()).thenReturn("Good condition");
        when(productDetails.getSellerRating()).thenReturn("4.5");
        String expectedOutput = "------------------- \n" + "SellerId = 12345 \n" + "SellerNickName = seller \n" + "ExchangeID = abcde \n" + "Price = 100$ \n" + "Condition = New \n" + "Condition Type = Used \n" + "Exchange Availability = Yes \n" + "County = USA \n" + "State = NY \n" + "Comments = Good condition \n" + "------------------- \n";
        assertEquals(expectedOutput, productDetails.toString());
    }
}
