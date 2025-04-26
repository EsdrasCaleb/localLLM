package com.densebrain.rif.server;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import com.densebrain.rif.server.transport.WebServiceContainer;

@ExtendWith(MockitoExtension.class)
public class RIFServer_stop_1_1_Test {

    @Mock
    private WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer = new RIFServer(8080);

    @Test
    void testStop_ContainerRunning() throws Exception {
        // Arrange
        when(container.isRunning()).thenReturn(true);
        // Act
        rifServer.stop();
        // Assert
        verify(container, times(1)).stopContainer();
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        assertNull(containerField.get(rifServer));
    }

    @Test
    void testStop_ContainerNotRunning() throws Exception {
        // Arrange
        when(container.isRunning()).thenReturn(false);
        // Act
        rifServer.stop();
        // Assert
        verify(container, never()).stopContainer();
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        assertNotNull(containerField.get(rifServer));
    }

    @Test
    void testStop_ContainerNull() throws Exception {
        // Arrange
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        containerField.set(rifServer, null);
        // Act
        rifServer.stop();
        // Assert
        verify(container, never()).stopContainer();
        // The container should remain null.  No need to verify again.
    }

    // Helper class for mocking - replace with your actual WebServiceContainer if available
    static class WebServiceContainer {

        private boolean running = true;

        public boolean isRunning() {
            return running;
        }

        public void stopContainer() {
            running = false;
        }
    }

    // Dummy RIFServer class - replace with your actual class
    static class RIFServer {

        private int port;

        private WebServiceContainer container;

        private static final Log LOG = LogFactory.getLog(RIFServer.class);

        public RIFServer(int port) {
            this.port = port;
            this.container = new WebServiceContainer();
        }

        public void stop() throws RemoteException {
            if (container != null && container.isRunning()) {
                container.stopContainer();
                container = null;
            }
        }
    }
}
