package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Accessories_getAccessory_3_0_Test {

    @Test
    void getAccessory() {
        Accessories accessories = new Accessories();
        accessories.setAccessory(new String[] { "accessory1", "accessory2", "accessory3" });
        assertEquals("accessory1", accessories.getAccessory(0));
        assertEquals("accessory2", accessories.getAccessory(1));
        assertEquals("accessory3", accessories.getAccessory(2));
    }
}
