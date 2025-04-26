package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Features_getFeature_3_0_Test {

    @Test
    public void testGetFeature() {
        Features features = new Features();
        features.setFeature(new String[] { "feature1", "feature2", "feature3" });
        String[] expected = { "feature1", "feature2", "feature3" };
        String actual = features.getFeature(0);
        assertEquals(expected, actual);
    }
}
