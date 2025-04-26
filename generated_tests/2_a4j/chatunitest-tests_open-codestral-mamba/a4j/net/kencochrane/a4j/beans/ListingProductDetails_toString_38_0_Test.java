package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ListingProductDetails_toString_38_0_Test {

    private ListingProductDetails listing;

    @BeforeEach
    public void setUp() {
        listing = Mockito.spy(new ListingProductDetails());
    }

    @Test
    public void testToString() {
        // Set up test data
        String exchangeId = "123";
        String listingId = "456";
        String exchangeTitle = "Test Title";
        String exchangePrice = "10.00";
        String exchangeAsin = "B001234567";
        String exchangeEndDate = "2022-12-31";
        String exchangeOfferingType = "Offer";
        String exchangeSellerId = "789";
        String exchangeSellerNickname = "Test Seller";
        String exchangeStartDate = "2022-01-01";
        String exchangeStatus = "Active";
        String exchangeQuantity = "10";
        String exchangeQuantityAllocated = "5";
        String exchangeFeaturedCategory = "Electronics";
        String exchangeConditionType = "New";
        String exchangeAvailability = "In Stock";
        String exchangeSellerState = "CA";
        String exchangeSellerCountry = "US";
        String exchangeSellerRating = "4.5";
        // Set up expected output
        String expectedOutput = " ----------- <br />\n" + "ASIN B001234567 <br />\n" + "Avail In Stock <br />\n" + "Condition Type New <br />\n" + "EndDate 2022-12-31 <br />\n" + "Featured Cat Electronics <br />\n" + "Ex ID 123 <br />\n" + "Offer Type Offer <br />\n" + "Ex Price 10.00 <br />\n" + "Ex Quant 10 <br />\n" + "Quantity Allocated 5 <br />\n" + "Seller Country US <br />\n" + "Seller Id 789 <br />\n" + "Seller Nickname Test Seller <br />\n" + "Seller Rating 4.5 <br />\n" + "Seller State CA <br />\n" + "Start date 2022-01-01 <br />\n" + "Status Active <br />\n" + "Title Test Title <br />\n" + " ----------- <br />\n";
        // Set up mocks
        Mockito.doReturn(exchangeId).when(listing).getExchangeId();
        Mockito.doReturn(listingId).when(listing).getListingId();
        Mockito.doReturn(exchangeTitle).when(listing).getExchangeTitle();
        Mockito.doReturn(exchangePrice).when(listing).getExchangePrice();
        Mockito.doReturn(exchangeAsin).when(listing).getExchangeAsin();
        Mockito.doReturn(exchangeEndDate).when(listing).getExchangeEndDate();
        Mockito.doReturn(exchangeOfferingType).when(listing).getExchangeOfferingType();
        Mockito.doReturn(exchangeSellerId).when(listing).getExchangeSellerId();
        Mockito.doReturn(exchangeSellerNickname).when(listing).getExchangeSellerNickname();
        Mockito.doReturn(exchangeStartDate).when(listing).getExchangeStartDate();
        Mockito.doReturn(exchangeStatus).when(listing).getExchangeStatus();
        Mockito.doReturn(exchangeQuantity).when(listing).getExchangeQuantity();
    }
}
