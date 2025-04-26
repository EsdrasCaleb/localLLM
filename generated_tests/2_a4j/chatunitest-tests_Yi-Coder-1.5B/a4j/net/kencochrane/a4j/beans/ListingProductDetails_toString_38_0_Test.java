package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ListingProductDetails_toString_38_0_Test {

    @Test
    void testToString() {
        ListingProductDetails listingProductDetails = new ListingProductDetails();
        listingProductDetails.setExchangeAsin("123456789");
        listingProductDetails.setExchangeAvailability("in stock");
        listingProductDetails.setExchangeConditionType("new");
        listingProductDetails.setExchangeEndDate("2023-01-01");
        listingProductDetails.setExchangeFeaturedCategory("Electronics");
        listingProductDetails.setExchangeId("123456789");
        listingProductDetails.setExchangeOfferingType("new");
        listingProductDetails.setExchangePrice("100.00");
        listingProductDetails.setExchangeQuantity("10");
        listingProductDetails.setExchangeQuantityAllocated("5");
        listingProductDetails.setExchangeSellerCountry("US");
        listingProductDetails.setExchangeSellerId("123456789");
        listingProductDetails.setExchangeSellerNickname("testnickname");
        listingProductDetails.setExchangeSellerRating("100");
        listingProductDetails.setExchangeSellerState("CA");
        listingProductDetails.setExchangeStartDate("2022-12-31");
        listingProductDetails.setExchangeStatus("active");
        listingProductDetails.setExchangeTitle("testtitle");
        String expectedOutput = " ----------- \n" + "ASIN 123456789\n" + "Avail in stock\n" + "Condition Type new\n" + "EndDate 2023-01-01\n" + "Featured Cat Electronics\n" + "Ex ID 123456789\n" + "Offer Type new\n" + "Ex Price 100.00\n" + "Quantity Allocated 5\n" + "Seller Country US\n" + "Seller Id 123456789\n" + "Seller Nickname testnickname\n" + "Seller Rating 100\n" + "Seller State CA\n" + "Start date 2022-12-31\n" + "Status active\n" + "Title testtitle\n" + " ----------- \n";
        String actualOutput = listingProductDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
