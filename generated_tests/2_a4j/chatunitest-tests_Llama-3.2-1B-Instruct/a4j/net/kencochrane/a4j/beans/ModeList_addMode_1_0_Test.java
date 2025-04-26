package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ModeList_addMode_1_0_Test {

    @Mock
    private Mode mode;

    @InjectMocks
    private ModeList modeList;

    @Test
    public void testAddMode() {
        // Arrange
        List<Mode> modes = new ArrayList<>();
        modeList.addMode(mode);
        // Act
        modeList.addMode(mode);
        // Assert
        assertEquals(1, modeList.modes.size());
    }
}
