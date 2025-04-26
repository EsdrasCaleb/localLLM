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

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private ThirdPartyProductDetails thirdPartyProductDetails;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        thirdPartyProductDetails.setSellerId("123");
        thirdPartyProductDetails.setSellerNickname("TestNickname");
        thirdPartyProductDetails.setExchangeId("456");
        thirdPartyProductDetails.setOfferingPrice("100");
        thirdPartyProductDetails.setCondition("Good");
        thirdPartyProductDetails.setConditionType("New");
        thirdPartyProductDetails.setExchangeAvailability("Available");
        thirdPartyProductDetails.setSellerCountry("USA");
        thirdPartyProductDetails.setSellerState("California");
        thirdPartyProductDetails.setShipComments("Test Comments");
        String expectedOutput = "------------------- \n" + "SellerId = 123 \n" + "SellerNickName = TestNickname \n" + "ExchangeID = 456 \n" + "Price =  100 \n" + "Condition = Good \n" + "Condition Type = New \n" + "Exchange Availability = Available \n" + "County = USA \n" + "State = California \n" + "Comments = Test Comments \n" + "------------------- \n";
        assertEquals(expectedOutput, thirdPartyProductDetails.toString());
    }
}
