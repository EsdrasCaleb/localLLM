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
        ListingProductDetails listingProduct = new ListingProductDetails();
        listingProduct.setExchangeId("12345");
        listingProduct.setListingId("67890");
        listingProduct.setExchangeTitle("Electronics");
        listingProduct.setExchangePrice("19.99");
        listingProduct.setExchangeAsin("B07X95K6Z9");
        listingProduct.setExchangeEndDate("2023-12-31");
        listingProduct.setExchangeOfferingType("New");
        listingProduct.setExchangeSellerId("Seller123");
        listingProduct.setExchangeSellerNickname("Seller1");
        listingProduct.setExchangeStartDate("2023-01-01");
        listingProduct.setExchangeStatus("Active");
        listingProduct.setExchangeQuantity("10");
        listingProduct.setExchangeQuantityAllocated("5");
        listingProduct.setExchangeFeaturedCategory("Electronics");
        listingProduct.setExchangeConditionType("New");
        listingProduct.setExchangeAvailability("In Stock");
        listingProduct.setExchangeSellerState("Active");
        listingProduct.setExchangeSellerCountry("USA");
        listingProduct.setExchangeSellerRating("4.5");
        String output = listingProduct.toString();
        assertEquals(" ----------- <br />\n", output);
        assertEquals("ASIN 12345\nAvail Active\nCondition Type New\nEndDate 2023-12-31\nFeatured Cat Electronics\nEx ID 12345\nOffer Type New\nEx Price 19.99\nEx Quant 10\nQuantity Allocated 5\nSeller Country USA\nSeller Id Seller123\nSeller Nickname Seller1\nSeller Rating 4.5\nSeller State Active\nStart date 2023-01-01\nStatus Active\nTitle Electronics\n ----------- <br />\n", output);
    }
}
