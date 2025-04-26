package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_0_Test {

    private SellerProfileDetails sellerProfileDetails;

    @BeforeEach
    void setUp() {
        sellerProfileDetails = new SellerProfileDetails();
        sellerProfileDetails.setSellerNickname("TestNickname");
        sellerProfileDetails.setOverallFeedbackRating("TestRating");
        sellerProfileDetails.setNumberOfFeedback("TestNumberOfFeedback");
        sellerProfileDetails.setNumberofCanceledAuctions("TestNumberOfCanceledAuctions");
        sellerProfileDetails.setStoreId("TestStoreId");
        sellerProfileDetails.setStoreName("TestStoreName");
        sellerProfileDetails.setSellerFeedBack(new SellerFeedback());
    }

    @Test
    void testToString() {
        String expectedToString = "NickName = TestNickname\n" + "OverallRating = TestRating\n" + "# of feedbacks = TestNumberOfFeedback\n" + "# of Canceled Auctions = TestNumberOfCanceledAuctions\n" + "StoreId = TestStoreId\n" + "StoreName = TestStoreName\n" + "FeedBack = \n" + new SellerFeedback() + "\n";
        assertEquals(expectedToString, sellerProfileDetails.toString());
    }
}
