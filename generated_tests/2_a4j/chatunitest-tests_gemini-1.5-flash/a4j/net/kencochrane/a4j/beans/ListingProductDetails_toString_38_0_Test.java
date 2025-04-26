package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ListingProductDetails_toString_38_0_Test {

    @Test
    void testToString() {
        ListingProductDetails details = new ListingProductDetails();
        details.setExchangeAsin("B012345678");
        details.setExchangeAvailability("Available");
        details.setExchangeConditionType("New");
        details.setExchangeEndDate("2024-12-31");
        details.setExchangeFeaturedCategory("Electronics");
        details.setExchangeId("12345");
        details.setExchangeOfferingType("BuyBox");
        details.setExchangePrice("99.99");
        details.setExchangeQuantity("100");
        details.setExchangeQuantityAllocated("50");
        details.setExchangeSellerCountry("US");
        details.setExchangeSellerId("seller123");
        details.setExchangeSellerNickname("SellerName");
        details.setExchangeSellerRating("4.5");
        details.setExchangeSellerState("CA");
        details.setExchangeStartDate("2023-10-26");
        details.setExchangeStatus("Active");
        details.setExchangeTitle("Test Product");
        String expectedOutput = " ----------- <br />\n" + "ASIN B012345678<br />\n" + "Avail Available<br />\n" + "Condition Type New<br />\n" + "EndDate 2024-12-31<br />\n" + "Featured Cat Electronics<br />\n" + "Ex ID 12345<br />\n" + "Offer Type BuyBox<br />\n" + "Ex Price 99.99<br />\n" + "Ex Quant 100<br />\n" + "Quantity Allocated 50<br />\n" + "Seller Country US<br />\n" + "Seller Id seller123<br />\n" + "Seller Nickname SellerName<br />\n" + "Seller Rating 4.5<br />\n" + "Seller State CA<br />\n" + "Start date 2023-10-26<br />\n" + "Status Active<br />\n" + "Title Test Product<br />\n" + " ----------- <br />\n";
        String actualOutput = details.toString();
        assertEquals(expectedOutput, actualOutput);
        // Test with null values
        ListingProductDetails details2 = new ListingProductDetails();
        String expectedOutput2 = " ----------- <br />\n" + "ASIN null<br />\n" + "Avail null<br />\n" + "Condition Type null<br />\n" + "EndDate null<br />\n" + "Featured Cat null<br />\n" + "Ex ID null<br />\n" + "Offer Type null<br />\n" + "Ex Price null<br />\n" + "Ex Quant null<br />\n" + "Quantity Allocated null<br />\n" + "Seller Country null<br />\n" + "Seller Id null<br />\n" + "Seller Nickname null<br />\n" + "Seller Rating null<br />\n" + "Seller State null<br />\n" + "Start date null<br />\n" + "Status null<br />\n" + "Title null<br />\n" + " ----------- <br />\n";
        String actualOutput2 = details2.toString();
        assertEquals(expectedOutput2, actualOutput2);
    }
}
