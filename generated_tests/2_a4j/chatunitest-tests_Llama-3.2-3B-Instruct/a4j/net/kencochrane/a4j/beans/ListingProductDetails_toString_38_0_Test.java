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
    public void testToString() {
        ListingProductDetails listingProductDetails = new ListingProductDetails();
        listingProductDetails.setExchangeAsin("XXXXXX");
        listingProductDetails.setExchangeAvailability("Available");
        listingProductDetails.setExchangeConditionType("New");
        listingProductDetails.setExchangeEndDate("2024-03-16");
        listingProductDetails.setExchangeFeaturedCategory("Electronics");
        listingProductDetails.setExchangeId("1234567890");
        listingProductDetails.setExchangeOfferingType("Fixed Price");
        listingProductDetails.setExchangePrice("$10.00");
        listingProductDetails.setExchangeQuantity("1");
        listingProductDetails.setExchangeQuantityAllocated("1");
        listingProductDetails.setExchangeSellerCountry("USA");
        listingProductDetails.setExchangeSellerId("seller123");
        listingProductDetails.setExchangeSellerNickname("seller123");
        listingProductDetails.setExchangeSellerRating("5.0");
        listingProductDetails.setExchangeSellerState("CA");
        listingProductDetails.setExchangeStartDate("2024-03-16");
        listingProductDetails.setExchangeStatus("Active");
        listingProductDetails.setExchangeTitle("Sony PlayStation 5");
        String expectedOutput = " ----------- <br />" + "ASIN XXXXXXXX<br />" + "Avail Available<br />" + "Condition Type New<br />" + "EndDate 2024-03-16<br />" + "Featured Cat Electronics<br />" + "Ex ID 1234567890<br />" + "Offer Type Fixed Price<br />" + "Ex Price $10.00<br />" + "Ex Quant 1<br />" + "Quantity Allocated 1<br />" + "Seller Country USA<br />" + "Seller Id seller123<br />" + "Seller Nickname seller123<br />" + "Seller Rating 5.0<br />" + "Seller State CA<br />" + "Start date 2024-03-16<br />" + "Status Active<br />" + "Title Sony PlayStation 5<br />" + " ----------- <br />";
        String actualOutput = listingProductDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
