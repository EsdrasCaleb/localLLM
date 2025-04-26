package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.rmi.RemoteException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

@ExtendWith(MockitoExtension.class)
public class RIFServer_start_0_2_Test {

    @Mock
    private WebServiceContainer mockContainer;

    @InjectMocks
    private RIFServer rifServer;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        containerField.set(rifServer, mockContainer);
    }

    @Test
    public void testStart() throws Exception {
        rifServer.start();
        verify(mockContainer, times(1)).startContainer();
    }

    @Test
    public void testStart_RemoteException() throws Exception {
        doThrow(new RemoteException()).when(mockContainer).startContainer();
        assertThrows(RemoteException.class, () -> rifServer.start());
    }
}
