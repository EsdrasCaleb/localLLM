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
        listingProductDetails.setExchangeAsin("testAsin");
        listingProductDetails.setExchangeAvailability("testAvailability");
        listingProductDetails.setExchangeConditionType("testConditionType");
        listingProductDetails.setExchangeEndDate("testEndDate");
        listingProductDetails.setExchangeFeaturedCategory("testFeaturedCategory");
        listingProductDetails.setExchangeId("testId");
        listingProductDetails.setExchangeOfferingType("testOfferingType");
        listingProductDetails.setExchangePrice("testPrice");
        listingProductDetails.setExchangeQuantity("testQuantity");
        listingProductDetails.setExchangeQuantityAllocated("testQuantityAllocated");
        listingProductDetails.setExchangeSellerCountry("testSellerCountry");
        listingProductDetails.setExchangeSellerId("testSellerId");
        listingProductDetails.setExchangeSellerNickname("testSellerNickname");
        listingProductDetails.setExchangeSellerRating("testSellerRating");
        listingProductDetails.setExchangeSellerState("testSellerState");
        listingProductDetails.setExchangeStartDate("testStartDate");
        listingProductDetails.setExchangeStatus("testStatus");
        listingProductDetails.setExchangeTitle("testTitle");
        String expected = " ----------- <br />\n" + "ASIN testAsin<br />\n" + "Avail testAvailability<br />\n" + "Condition Type testConditionType<br />\n" + "EndDate testEndDate<br />\n" + "Featured Cat testFeaturedCategory<br />\n" + "Ex ID testId<br />\n" + "Offer Type testOfferingType<br />\n" + "Ex Price testPrice<br />\n" + "Ex Quant testQuantity<br />\n" + "Quantity Allocated testQuantityAllocated<br />\n" + "Seller Country testSellerCountry<br />\n" + "Seller Id testSellerId<br />\n" + "Seller Nickname testSellerNickname<br />\n" + "Seller Rating testSellerRating<br />\n" + "Seller State testSellerState<br />\n" + "Start date testStartDate<br />\n" + "Status testStatus<br />\n" + "Title testTitle<br />\n" + " ----------- <br />\n";
        assertEquals(expected, listingProductDetails.toString());
    }
}
