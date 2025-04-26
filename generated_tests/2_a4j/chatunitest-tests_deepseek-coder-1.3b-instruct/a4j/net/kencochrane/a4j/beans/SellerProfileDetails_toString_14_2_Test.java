package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_2_Test {

    @Test
    public void testToString() {
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        sellerProfileDetails.setSellerNickname("testNickname");
        sellerProfileDetails.setOverallFeedbackRating("testRating");
        sellerProfileDetails.setNumberOfFeedback("testFeedback");
        sellerProfileDetails.setNumberofCanceledAuctions("testCanceledAuctions");
        sellerProfileDetails.setStoreId("testStoreId");
        sellerProfileDetails.setStoreName("testStoreName");
        String expectedOutput = "NickName = testNickname\n" + "OverallRating = testRating\n" + "# of feedbacks = testFeedback\n" + "# of Canceled Auctions = testCanceledAuctions\n" + "StoreId = testStoreId\n" + "StoreName = testStoreName\n" + "FeedBack = \n";
        assertEquals(expectedOutput, sellerProfileDetails.toString());
    }
}
