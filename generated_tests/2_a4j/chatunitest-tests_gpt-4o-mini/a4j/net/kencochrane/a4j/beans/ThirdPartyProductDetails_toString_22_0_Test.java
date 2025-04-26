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

    private ThirdPartyProductDetails productDetails;

    @BeforeEach
    public void setUp() {
        productDetails = new ThirdPartyProductDetails();
        productDetails.setSellerId("12345");
        productDetails.setSellerNickname("BestSeller");
        productDetails.setExchangeId("EX123");
        productDetails.setOfferingPrice("99.99");
        productDetails.setCondition("New");
        productDetails.setConditionType("Retail");
        productDetails.setExchangeAvailability("Available");
        productDetails.setSellerCountry("USA");
        productDetails.setSellerState("CA");
        productDetails.setShipComments("Ships within 24 hours");
        productDetails.setSellerRating("4.5");
    }

    @Test
    public void testToString() {
        String expectedOutput = "------------------- \n" + "SellerId = 12345\n" + "SellerNickName = BestSeller\n" + "ExchangeID = EX123\n" + "Price =  99.99\n" + "Condition = New\n" + "Condition Type = Retail\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = CA\n" + "Comments = Ships within 24 hours\n" + "------------------- \n";
        assertEquals(expectedOutput, productDetails.toString());
    }
}
