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

public class ThirdPartyProductDetails_toString_22_0_Test {

    private ThirdPartyProductDetails productDetails;

    @BeforeEach
    void setUp() {
        productDetails = new ThirdPartyProductDetails();
    }

    @Test
    void testToString() throws Exception {
        // Setting up the private fields using reflection
        setField(productDetails, "sellerId", "Seller123");
        setField(productDetails, "sellerNickname", "NickName");
        setField(productDetails, "exchangeId", "Exch123");
        setField(productDetails, "offeringPrice", "100.00");
        setField(productDetails, "condition", "New");
        setField(productDetails, "conditionType", "Brand New");
        setField(productDetails, "exchangeAvailability", "Available");
        setField(productDetails, "sellerCountry", "USA");
        setField(productDetails, "sellerState", "CA");
        setField(productDetails, "sellerRating", "5.0");
        setField(productDetails, "shipComments", "No comments");
        // Expected output string
        String expectedOutput = "------------------- \n" + "SellerId = Seller123\n" + "SellerNickName = NickName\n" + "ExchangeID = Exch123\n" + "Price =  100.00\n" + "Condition = New\n" + "Condition Type = Brand New\n" + "Exchange Availability = Available\n" + "County = USA\n" + "State = CA\n" + "Comments = No comments\n" + "------------------- \n";
        // Actual output from the toString method
        String actualOutput = productDetails.toString();
        // Assertion to check if the actual output matches the expected output
        assertEquals(expectedOutput, actualOutput);
    }

    private void setField(Object object, String fieldName, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(object, value);
    }
}
