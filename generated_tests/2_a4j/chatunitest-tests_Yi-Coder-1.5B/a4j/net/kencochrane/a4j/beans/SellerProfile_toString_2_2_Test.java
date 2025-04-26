package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfile_toString_2_2_Test {

    private SellerProfile sellerProfile;

    @BeforeEach
    void setUp() {
        sellerProfile = new SellerProfile();
    }

    @Test
    void testToString() {
        String expected = sellerProfile.getSellerProfileDetails().toString() + "\n";
        String actual = sellerProfile.toString();
        Assertions.assertEquals(expected, actual);
    }
}
