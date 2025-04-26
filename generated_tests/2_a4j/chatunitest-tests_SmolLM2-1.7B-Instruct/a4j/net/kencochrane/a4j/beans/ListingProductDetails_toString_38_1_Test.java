package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ListingProductDetails_toString_38_1_Test {

    @Test
    public void testToString() {
        ListingProductDetails listingProductDetails = new ListingProductDetails();
        listingProductDetails.setExchangeId("123456789");
        listingProductDetails.setExchangeTitle("Product Title");
        listingProductDetails.setExchangePrice("100.00");
        listingProductDetails.setExchangeAsin("ASIN123456789");
        listingProductDetails.setExchangeEndDate("2022-01-01");
        listingProductDetails.setExchangeOfferingType("OfferType");
        listingProductDetails.setExchangeQuantity("10");
        listingProductDetails.setExchangeQuantityAllocated("5");
        listingProductDetails.setExchangeFeaturedCategory("Featured Category");
        listingProductDetails.setExchangeConditionType("Condition Type");
        listingProductDetails.setExchangeAvailability("Avail");
        listingProductDetails.setExchangeSellerCountry("Country");
        listingProductDetails.setExchangeSellerId("SellerId");
        listingProductDetails.setExchangeSellerNickname("SellerNickname");
        listingProductDetails.setExchangeSellerRating("SellerRating");
        listingProductDetails.setExchangeStartDate("2022-01-01");
        listingProductDetails.setExchangeStatus("Status");
        listingProductDetails.setExchangeTitle("Product Title");
        assertEquals(listingProductDetails.toString(), " ----------- <br />\nASIN ASIN123456789<br />\nAvail Avail<br />\nCondition Type Condition Type<br />\nEndDate 2022-01-01<br />\nFeatured Cat Featured Category<br />\nEx ID 123456789<br />\nOffer Type OfferType<br />\nEx Price 100.00<br />\nEx Quant 10<br />\nQuantity Allocated 5<br />\nSeller Country Country<br />\nSeller Id SellerId<br />\nSeller Nickname SellerNickname<br />\nSeller Rating SellerRating<br />\nSeller State Country<br />\nStart date 2022-01-01<br />\nStatus Status<br />\nTitle Product Title<br />\n ----------- <br />\n");
    }
}
