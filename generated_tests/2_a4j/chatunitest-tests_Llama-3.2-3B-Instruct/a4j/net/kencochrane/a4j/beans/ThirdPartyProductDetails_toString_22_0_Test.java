package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ThirdPartyProductDetails_toString_22_0_Test {

    @Mock
    private a4jUtil a4jUtil;

    private ThirdPartyProductDetails details;

    @BeforeEach
    public void setup() {
        details = new ThirdPartyProductDetails();
    }

    @Test
    public void testToString() {
        // Arrange
        details.setSellerId("123");
        details.setSellerNickname("John Doe");
        details.setExchangeId("ABC");
        details.setOfferingPrice("100.00");
        details.setCondition("Good");
        details.setConditionType("New");
        details.setExchangeAvailability("Available");
        details.setSellerCountry("USA");
        details.setSellerState("NY");
        details.setSellerRating("5");
        details.setShipComments("Fast shipping");
        // Act
        String expectedOutput = "-------------------\n" + "SellerId = 123\n" + "SellerNickName = John Doe\n" + "ExchangeID = ABC\n" + "Price =  100.00\n" + "Condition = Good\n" + "Condition Type = New\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = NY\n" + "Comments = Fast shipping\n" + "-------------------";
        // Assert
        String actualOutput = details.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_EmptyFields() {
        // Arrange
        details.setSellerId("");
        details.setSellerNickname("");
        details.setExchangeId("");
        details.setOfferingPrice("");
        details.setCondition("");
        details.setConditionType("");
        details.setExchangeAvailability("");
        details.setSellerCountry("");
        details.setSellerState("");
        details.setSellerRating("");
        details.setShipComments("");
        // Act
        String actualOutput = details.toString();
        // Assert
        assertTrue(actualOutput.contains("-------------------"));
        assertTrue(actualOutput.contains("SellerId = "));
        assertTrue(actualOutput.contains("SellerNickName = "));
        assertTrue(actualOutput.contains("ExchangeID = "));
        assertTrue(actualOutput.contains("Price =  "));
        assertTrue(actualOutput.contains("Condition = "));
        assertTrue(actualOutput.contains("Condition Type = "));
        assertTrue(actualOutput.contains("Exchange Availability = "));
        assertTrue(actualOutput.contains("County = "));
        assertTrue(actualOutput.contains("State = "));
        assertTrue(actualOutput.contains("Comments = "));
        assertTrue(actualOutput.contains("-------------------"));
    }
}
