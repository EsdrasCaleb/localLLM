package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_2_Test {

    @Test
    void testToString() {
        SellerProfile sellerProfile = new SellerProfile();
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        sellerProfile.setSellerProfileDetails(sellerProfileDetails);
        String expectedOutput = "SellerProfile Details:\nName: John Doe\nContact: john.doe@example.com\nAddress: 123 Main St\n";
        String actualOutput = sellerProfile.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
