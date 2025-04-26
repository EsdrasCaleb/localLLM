package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ListingProductDetails_toString_38_0_Test {

    @Mock
    private ListingProductDetails focal;

    @InjectMocks
    private ListingProductDetails testingInstance;

    @Test
    public void testToString() {
        // Arrange
        when(focal.getExchangeAsin()).thenReturn("ASIN123");
        when(focal.getExchangeAvailability()).thenReturn("Available");
        when(focal.getExchangeConditionType()).thenReturn("Condition Type");
        when(focal.getExchangeSellerCountry()).thenReturn("Country");
        when(focal.getExchangeSellerState()).thenReturn("State");
        when(focal.getExchangeSellerRating()).thenReturn("Rating");
        // Act
        String output = testingInstance.toString();
        // Assert
        String expectedOutput = " ----------- <br />\n" + "ASIN ASIN123<br />\n" + "Avail Available<br />\n" + "Condition Type Condition Type<br />\n" + "EndDate 2021-01-01<br />\n" + "Featured Cat Condition Type<br />\n" + "Ex ID ASIN123<br />\n" + "Offer Type Offer Type<br />\n" + "Ex Price 1000.00<br />\n" + "Ex Quant 1000<br />\n" + "Quantity Allocated 1000<br />\n" + "Seller Country Country<br />\n" + "Seller Id ASIN123<br />\n" + "Seller Nickname Nickname<br />\n" + "Seller Rating Rating<br />\n" + "Seller State State<br />\n" + "Start date 2021-01-01<br />\n" + "Status Available<br />\n" + "Title Title<br />\n" + " ----------- <br />\n";
        assertEquals(expectedOutput, output);
    }
}
