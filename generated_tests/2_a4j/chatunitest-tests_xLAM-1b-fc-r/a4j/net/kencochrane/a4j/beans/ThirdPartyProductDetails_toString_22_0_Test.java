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
    public void testToString() {
        // Arrange
        ThirdPartyProductDetails productDetails = mock(ThirdPartyProductDetails.class);
        when(productDetails.getSellerId()).thenReturn("12345");
        when(productDetails.getSellerNickname()).thenReturn("JohnDoe");
        when(productDetails.getExchangeId()).thenReturn("ABC123");
        when(productDetails.getOfferingPrice()).thenReturn("100.00");
        when(productDetails.getCondition()).thenReturn("Good");
        when(productDetails.getConditionType()).thenReturn("New");
        when(productDetails.getExchangeAvailability()).thenReturn("Yes");
        when(productDetails.getSellerCountry()).thenReturn("USA");
        when(productDetails.getSellerState()).thenReturn("New York");
        when(productDetails.getShipComments()).thenReturn("No comments");
        when(productDetails.getSellerRating()).thenReturn("4.5");
        // Act
        String expected = "------------------- \n" + "SellerId = " + 12345 + "\n" + "SellerNickName = " + "JohnDoe" + "\n" + "ExchangeID = " + "ABC123" + "\n" + "Price =  " + 100.00 + "\n" + "Condition = " + "Good" + "\n" + "Condition Type = " + "New" + "\n" + "Exchange Availability = " + "Yes" + "\n" + "County = " + "USA" + "\n" + "State = " + "New York" + "\n" + "Comments = " + "No comments" + "\n" + "------------------- \n";
        String actual = productDetails.toString();
        // Assert
        assertEquals(expected, actual);
    }
}
