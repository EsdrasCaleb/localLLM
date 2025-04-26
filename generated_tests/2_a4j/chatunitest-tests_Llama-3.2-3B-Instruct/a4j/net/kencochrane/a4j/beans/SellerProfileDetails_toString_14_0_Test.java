package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_0_Test {

    @Test
    public void testToString() throws Exception {
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        sellerProfileDetails.setSellerNickname("JohnDoe");
        sellerProfileDetails.setOverallFeedbackRating("5");
        sellerProfileDetails.setNumberOfFeedback("10");
        sellerProfileDetails.setNumberofCanceledAuctions("2");
        sellerProfileDetails.setStoreId("123");
        sellerProfileDetails.setStoreName("ABC Store");
        sellerProfileDetails.setSellerFeedBack(new SellerFeedback());
        Method method = SellerProfileDetails.class.getDeclaredMethod("toString");
        method.setAccessible(true);
        String expectedOutput = "NickName = JohnDoe\n" + "OverallRating = 5\n" + "# of feedbacks = 10\n" + "# of Canceled Auctions = 2\n" + "StoreId = 123\n" + "StoreName = ABC Store\n" + "FeedBack = \n" + new SellerFeedback().toString();
        String actualOutput = (String) method.invoke(sellerProfileDetails);
        assertEquals(expectedOutput, actualOutput);
        assertTrue(actualOutput.contains("JohnDoe"));
        assertTrue(actualOutput.contains("5"));
        assertTrue(actualOutput.contains("10"));
        assertTrue(actualOutput.contains("2"));
        assertTrue(actualOutput.contains("123"));
        assertTrue(actualOutput.contains("ABC Store"));
    }
}
