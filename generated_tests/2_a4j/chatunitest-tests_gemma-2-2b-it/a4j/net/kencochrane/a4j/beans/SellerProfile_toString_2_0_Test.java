package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_0_Test {

    @Test
    void testToString() {
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(new SellerProfileDetails());
        String expected = "SellerProfileDetails\n";
        String actual = sellerProfile.toString();
        assertEquals(expected, actual);
    }
}
