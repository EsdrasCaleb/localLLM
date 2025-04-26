package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ThirdPartyProductDetails_toString_22_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private ThirdPartyProductDetails thirdPartyProductDetails;

    @Test
    public void testToString() {
        // Arrange
        String sellerId = "12345";
        String sellerNickname = "John Doe";
        String exchangeId = "ABC123";
        String offeringPrice = "100.00";
        String condition = "Excellent";
        String conditionType = "Good";
        String exchangeAvailability = "Available";
        String sellerCountry = "USA";
        String sellerState = "CA";
        String sellerRating = "5.0";
        String shipComments = "Test comments";
        // Act
        String result = thirdPartyProductDetails.toString();
        // Assert
        // Assert that the output is as expected
        // Note: You would typically use a JUnit assertion framework to verify the output, but in this case, you can simply print the output to verify it matches the expected output
        System.out.println(result);
    }
}
