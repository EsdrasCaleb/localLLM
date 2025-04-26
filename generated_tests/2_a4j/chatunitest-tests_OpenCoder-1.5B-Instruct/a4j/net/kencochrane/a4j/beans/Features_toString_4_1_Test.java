package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_toString_4_1_Test {

    private Features features;

    @Test
    public void testToString() {
        features = new Features();
        features.setFeature(new String[] { "Feature1", "Feature2" });
        assertEquals("Feature1\nFeature2\n", features.toString());
    }
}
