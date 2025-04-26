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
        sellerProfileDetails.setSellerNickname("TestSeller");
        sellerProfileDetails.setOverallFeedbackRating("4.5");
        sellerProfileDetails.setNumberOfFeedback("100");
        sellerProfileDetails.setNumberofCanceledAuctions("5");
        sellerProfileDetails.setStoreId("Store123");
        sellerProfileDetails.setStoreName("Test Store");
        // Assuming a default constructor exists
        SellerFeedback sellerFeedback = new SellerFeedback();
        sellerProfileDetails.setSellerFeedBack(sellerFeedback);
    }

    @Test
    void testToString() {
        String expectedOutput = "NickName = TestSeller\n" + "OverallRating = 4.5\n" + "# of feedbacks = 100\n" + "# of Canceled Auctions = 5\n" + "StoreId = Store123\n" + "StoreName = Test Store\n" + "FeedBack = \n" + sellerProfileDetails.getSellerFeedBack() + "\n";
        String actualOutput = sellerProfileDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToStringWithNullValues() {
        sellerProfileDetails.setSellerNickname(null);
        sellerProfileDetails.setOverallFeedbackRating(null);
        sellerProfileDetails.setNumberOfFeedback(null);
        sellerProfileDetails.setNumberofCanceledAuctions(null);
        sellerProfileDetails.setStoreId(null);
        sellerProfileDetails.setStoreName(null);
        String expectedOutput = "NickName = null\n" + "OverallRating = null\n" + "# of feedbacks = null\n" + "# of Canceled Auctions = null\n" + "StoreId = null\n" + "StoreName = null\n" + "FeedBack = \n" + sellerProfileDetails.getSellerFeedBack() + "\n";
        String actualOutput = sellerProfileDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
