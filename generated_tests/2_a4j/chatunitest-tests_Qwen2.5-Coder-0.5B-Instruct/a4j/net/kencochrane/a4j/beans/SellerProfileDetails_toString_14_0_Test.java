package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_0_Test {

    @Test
    public void testToString() {
        // Create an instance of SellerProfileDetails
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        // Set the values of the fields
        sellerProfileDetails.setSellerNickname("John Doe");
        sellerProfileDetails.setOverallFeedbackRating("5 out of 5");
        sellerProfileDetails.setNumberOfFeedback("10");
        sellerProfileDetails.setNumberofCanceledAuctions("2");
        sellerProfileDetails.setStoreId("12345");
        sellerProfileDetails.setStoreName("Example Store");
        sellerProfileDetails.setSellerFeedBack(new SellerFeedback());
        // Use reflection to invoke the toString() method
        String expectedOutput = "NickName = John Doe\n" + "OverallRating = 5 out of 5\n" + "# of feedbacks = 10\n" + "# of Canceled Auctions = 2\n" + "StoreId = 12345\n" + "StoreName = Example Store\n" + "FeedBack =\n" + "SellerFeedback: Feedback content goes here\n";
        // Call the toString() method using reflection
        String actualOutput = sellerProfileDetails.toString();
        // Verify that the actual output matches the expected output
        assertEquals(expectedOutput, actualOutput);
    }
}
