package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

class ThirdPartyProductDetails_toString_22_3_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private ThirdPartyProductDetails thirdPartyProductDetails;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToString() {
        // Arrange
        thirdPartyProductDetails.setSellerId("123");
        thirdPartyProductDetails.setSellerNickname("John Doe");
        thirdPartyProductDetails.setExchangeId("ABC123");
        thirdPartyProductDetails.setOfferingPrice("100.99");
        thirdPartyProductDetails.setCondition("New");
        thirdPartyProductDetails.setConditionType("Good");
        thirdPartyProductDetails.setExchangeAvailability("Available");
        thirdPartyProductDetails.setSellerCountry("USA");
        thirdPartyProductDetails.setSellerState("California");
        thirdPartyProductDetails.setShipComments("Free shipping");
        // Act
        String result = thirdPartyProductDetails.toString();
        // Assert
        assertEquals("------------------- \n" + "SellerId = 123\n" + "SellerNickName = John Doe\n" + "ExchangeID = ABC123\n" + "Price =  100.99\n" + "Condition = New\n" + "Condition Type = Good\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = California\n" + "Comments = Free shipping\n" + "------------------- \n", result);
    }
}
