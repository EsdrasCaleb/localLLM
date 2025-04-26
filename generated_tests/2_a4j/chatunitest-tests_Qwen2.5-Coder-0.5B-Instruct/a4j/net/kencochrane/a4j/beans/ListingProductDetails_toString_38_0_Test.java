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
    public void testToString() {
        ListingProductDetails listingProductDetails = new ListingProductDetails();
        // Verify that the toString() method returns a meaningful string
        assertEquals(" ----------- <br />\n" + "ASIN " + listingProductDetails.getExchangeAsin() + "<br />\n" + "Avail " + listingProductDetails.getExchangeAvailability() + "<br />\n" + "Condition Type " + listingProductDetails.getExchangeConditionType() + "<br />\n" + "EndDate " + listingProductDetails.getExchangeEndDate() + "<br />\n" + "Featured Cat " + listingProductDetails.getExchangeFeaturedCategory() + "<br />\n" + "Ex ID " + listingProductDetails.getExchangeId() + "<br />\n" + "Offer Type " + listingProductDetails.getExchangeOfferingType() + "<br />\n" + "Ex Price " + listingProductDetails.getExchangePrice() + "<br />\n" + "Ex Quantity " + listingProductDetails.getExchangeQuantity() + "<br />\n" + "Quantity Allocated " + listingProductDetails.getExchangeQuantityAllocated() + "<br />\n" + "Seller Country " + listingProductDetails.getExchangeSellerCountry() + "<br />\n" + "Seller ID " + listingProductDetails.getExchangeSellerId() + "<br />\n" + "Seller Nickname " + listingProductDetails.getExchangeSellerNickname() + "<br />\n" + "Seller Rating " + listingProductDetails.getExchangeSellerRating() + "<br />\n" + "Seller State " + listingProductDetails.getExchangeSellerState() + "<br />\n" + "Start date " + listingProductDetails.getExchangeStartDate() + "<br />\n" + "Status " + listingProductDetails.getExchangeStatus() + "<br />\n" + "Title " + listingProductDetails.getExchangeTitle() + "<br />\n" + " ----------- <br />\n", listingProductDetails.toString());
    }
}
