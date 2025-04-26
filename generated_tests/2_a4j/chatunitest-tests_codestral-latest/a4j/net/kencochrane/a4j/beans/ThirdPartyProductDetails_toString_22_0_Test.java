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
        MockitoAnnotations.openMocks(this);
        thirdPartyProductDetails.setSellerId("S123");
        thirdPartyProductDetails.setSellerNickname("SellerA");
        thirdPartyProductDetails.setExchangeId("E456");
        thirdPartyProductDetails.setOfferingPrice("100.00");
        thirdPartyProductDetails.setCondition("New");
        thirdPartyProductDetails.setConditionType("TypeA");
        thirdPartyProductDetails.setExchangeAvailability("Available");
        thirdPartyProductDetails.setSellerCountry("USA");
        thirdPartyProductDetails.setSellerState("CA");
        thirdPartyProductDetails.setShipComments("Fast shipping");
        thirdPartyProductDetails.setSellerRating("4.5");
    }

    @Test
    public void testToString() {
        String expected = "------------------- \n" + "SellerId = S123\n" + "SellerNickName = SellerA\n" + "ExchangeID = E456\n" + "Price =  100.00\n" + "Condition = New\n" + "Condition Type = TypeA\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = CA\n" + "Comments = Fast shipping\n" + "------------------- \n";
        String actual = thirdPartyProductDetails.toString();
        assertEquals(expected, actual);
    }
}
